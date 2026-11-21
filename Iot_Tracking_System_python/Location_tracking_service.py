import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
import logging
import math
import time
import threading

logger = logging.getLogger(__name__)

# Mean Earth radius in metres (used by the haversine formula)
EARTH_RADIUS_M = 6_371_000.0

# Column layout of the duplicate-event and geofence-alert output files
DUPLICATE_LOG_COLUMNS = ['processing_time', 'device_id', 'event_timestamp', 'latitude',
                         'longitude', 'sts', 'speed']
ALERT_LOG_COLUMNS = ['alert_time', 'alert_type', 'device_id', 'geofence_id', 'geofence_name',
                     'event_timestamp', 'previous_event_timestamp', 'latitude', 'longitude',
                     'distance_to_boundary_m']


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in metres between two points given in decimal degrees,
    computed with the haversine formula:
        a = sin²(Δφ/2) + cos φ1 · cos φ2 · sin²(Δλ/2)
        d = 2R · asin(√a)
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    # min() guards against floating-point rounding pushing a slightly above 1
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


@dataclass(frozen=True)
class Geofence:
    """Circular geographical boundary: centre point (lat/lon) plus radius in metres."""
    geofence_id: str
    name: str
    latitude: float
    longitude: float
    radius_m: float

    def __post_init__(self):
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Geofence '{self.geofence_id}': latitude must be within [-90, 90], got {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Geofence '{self.geofence_id}': longitude must be within [-180, 180], got {self.longitude}")
        if not (math.isfinite(self.radius_m) and self.radius_m > 0):
            raise ValueError(f"Geofence '{self.geofence_id}': radius_m must be a positive number, got {self.radius_m}")

    def distance_to_boundary(self, latitude: float, longitude: float) -> float:
        """
        Signed distance in metres from a point to the geofence boundary:
        negative = inside (that far from the edge), positive = outside, 0 = on the edge.
        """
        return haversine_distance(self.latitude, self.longitude, latitude, longitude) - self.radius_m

    def contains(self, latitude: float, longitude: float) -> bool:
        """True if the point is inside the geofence (a point on the boundary counts as inside)."""
        return self.distance_to_boundary(latitude, longitude) <= 0


class IoTLocationService:
    """
    Real-time IoT Location Tracking Service
    Handles out-of-order streaming data and maintains latest location per device
    """
    def __init__(self, output_file: str = "location_output2.csv"):
        self.latest_state: Dict[str, dict] = {}

        # Historical log
        self.history_log: list = []

        # Output file path
        self.output_file = output_file

        # Performance metrics
        self.processing_times: list = []
        self.total_events_processed = 0
        self.lock = threading.Lock()

        # Duplicate detection: an event is uniquely identified by (device_fk_id, time_stamp)
        self.seen_event_keys: Set[Tuple[str, pd.Timestamp]] = set()
        self.duplicate_events_count = 0
        self.duplicate_count_by_device: Dict[str, int] = {}
        self.duplicate_log: list = []

        # Geofencing
        self.geofences: Dict[str, Geofence] = {}
        # Separate inside/outside state for every (device_id, geofence_id) combination
        self.geofence_states: Dict[Tuple[str, str], dict] = {}
        self.alert_log: list = []

    def process_event(self, event: dict, current_time: datetime) -> bool:
        """
        Process a single location event
        Returns True if state was updated, False otherwise
        """
        start_time = time.time()
        device_id = str(event['device_fk_id'])
        event_timestamp = pd.to_datetime(event['time_stamp'])
        latitude = event['latitude']
        longitude = event['longitude']
        sts = pd.to_datetime(event['sts'])
        speed = event.get('speed', 0)
        updated = False

        with self.lock:
            # Duplicate detection: same device + same event time_stamp = same event.
            # Check-and-mark happens under the lock so two threads can't both accept it.
            event_key = (device_id, event_timestamp)
            if event_key in self.seen_event_keys:
                # Duplicate: never touches latest_state, history_log or geofence state/alerts
                self._record_duplicate(device_id, event_timestamp, latitude, longitude,
                                       sts, speed, current_time)
            else:
                self.seen_event_keys.add(event_key)

                # Check if this is the latest data for this device
                if device_id not in self.latest_state:
                    self.latest_state[device_id] = {
                        'timestamp': event_timestamp,
                        'latitude': latitude,
                        'longitude': longitude,
                        'sts': sts,
                        'speed': speed,
                        'last_updated_at': current_time
                    }
                    updated = True
                else:
                    # Compare with existing latest
                    current_latest = self.latest_state[device_id]['timestamp']
                    if event_timestamp > current_latest:
                        self.latest_state[device_id] = {
                            'timestamp': event_timestamp,
                            'latitude': latitude,
                            'longitude': longitude,
                            'sts': sts,
                            'speed': speed,
                            'last_updated_at': current_time
                        }
                        updated = True

                # Geofences are only evaluated when the device's latest location moves forward.
                # Late (out-of-order) events describe where the device *was*, so evaluating
                # them would produce false entry/exit alerts.
                if updated:
                    self._evaluate_geofences(device_id, latitude, longitude,
                                             event_timestamp, current_time)

                # Log to history
                self.history_log.append({
                    'processing_time': current_time,
                    'device_id': device_id,
                    'event_timestamp': event_timestamp,
                    'latitude': latitude,
                    'longitude': longitude,
                    'sts': sts,
                    'speed': speed,
                    'is_latest': updated
                })

            self.total_events_processed += 1

        # Record processing time
        processing_time = (time.time() - start_time) * 1000
        self.processing_times.append(processing_time)

        return updated

    # ------------------------------------------------------------------
    # Duplicate event detection
    # ------------------------------------------------------------------
    def _record_duplicate(self, device_id: str, event_timestamp, latitude: float, longitude: float,
                          sts, speed, current_time: datetime):
        """Count and log a rejected duplicate event. Caller must hold self.lock."""
        self.duplicate_events_count += 1
        self.duplicate_count_by_device[device_id] = self.duplicate_count_by_device.get(device_id, 0) + 1
        self.duplicate_log.append({
            'processing_time': current_time,
            'device_id': device_id,
            'event_timestamp': event_timestamp,
            'latitude': latitude,
            'longitude': longitude,
            'sts': sts,
            'speed': speed
        })
        logger.warning(
            "Duplicate event ignored: device=%s event_timestamp=%s sts=%s "
            "(device duplicates=%d, total duplicates=%d)",
            device_id, event_timestamp, sts,
            self.duplicate_count_by_device[device_id], self.duplicate_events_count
        )

    # ------------------------------------------------------------------
    # Geofencing
    # ------------------------------------------------------------------
    def add_geofence(self, geofence_id: str, name: str, latitude: float, longitude: float,
                     radius_m: float) -> Geofence:
        """
        Define a circular geofence. Multiple geofences are supported.
        Args:
            geofence_id: Unique geofence identifier
            name: Human-readable name
            latitude, longitude: Centre coordinates in decimal degrees
            radius_m: Radius in metres
        Raises:
            ValueError: if the id already exists or the coordinates/radius are invalid
        """
        geofence = Geofence(str(geofence_id), name, float(latitude), float(longitude), float(radius_m))
        with self.lock:
            if geofence.geofence_id in self.geofences:
                raise ValueError(f"Geofence '{geofence.geofence_id}' already exists")
            self.geofences[geofence.geofence_id] = geofence
        logger.info("Geofence added: id=%s name='%s' center=(%.6f, %.6f) radius=%.1f m",
                    geofence.geofence_id, geofence.name, geofence.latitude, geofence.longitude,
                    geofence.radius_m)
        return geofence

    def _evaluate_geofences(self, device_id: str, latitude: float, longitude: float,
                            event_timestamp, current_time: datetime):
        """
        Update the state of every (device, geofence) pair from the device's new latest
        location and raise an alert only when the inside/outside state flips.
        The first location seen for a pair sets the baseline without an alert, because
        no entry or exit has actually been observed yet. Caller must hold self.lock.
        """
        for geofence in self.geofences.values():
            distance_to_boundary = geofence.distance_to_boundary(latitude, longitude)
            inside = distance_to_boundary <= 0
            state_key = (device_id, geofence.geofence_id)
            previous = self.geofence_states.get(state_key)

            self.geofence_states[state_key] = {
                'inside': inside,
                'distance_to_boundary_m': distance_to_boundary,
                'event_timestamp': event_timestamp,
                'latitude': latitude,
                'longitude': longitude
            }

            if previous is None:
                logger.info("Geofence baseline: device %s is %s '%s' (%s) at %s, %.1f m from boundary",
                            device_id, 'INSIDE' if inside else 'OUTSIDE', geofence.name,
                            geofence.geofence_id, event_timestamp, abs(distance_to_boundary))
                continue

            if previous['inside'] == inside:
                continue  # still inside / still outside -> no repeated alert

            alert_type = 'ENTRY' if inside else 'EXIT'
            self.alert_log.append({
                'alert_time': current_time,
                'alert_type': alert_type,
                'device_id': device_id,
                'geofence_id': geofence.geofence_id,
                'geofence_name': geofence.name,
                'event_timestamp': event_timestamp,
                'previous_event_timestamp': previous['event_timestamp'],
                'latitude': latitude,
                'longitude': longitude,
                'distance_to_boundary_m': distance_to_boundary
            })
            logger.info("GEOFENCE %s: device %s %s '%s' (%s) at %s (%.6f, %.6f), %.1f m %s boundary",
                        alert_type, device_id, 'entered' if inside else 'exited', geofence.name,
                        geofence.geofence_id, event_timestamp, latitude, longitude,
                        abs(distance_to_boundary), 'inside' if inside else 'outside')

    def get_device_geofence_status(self, device_id: str) -> List[dict]:
        """
        Check whether a device is inside or outside each geofence, based on its
        latest known location. Returns an empty list if the device is unknown.
        """
        device_id = str(device_id)
        with self.lock:
            state = self.latest_state.get(device_id)
            if state is None:
                return []
            status = []
            for geofence in self.geofences.values():
                distance_to_boundary = geofence.distance_to_boundary(state['latitude'], state['longitude'])
                status.append({
                    'geofence_id': geofence.geofence_id,
                    'geofence_name': geofence.name,
                    'inside': distance_to_boundary <= 0,
                    'distance_to_boundary_m': round(distance_to_boundary, 2),
                    'as_of': state['timestamp']
                })
            return status

    def stream_from_csv(self, csv_file: str, time_scale: float = 0.001):
        """
        Simulate streaming data from CSV based on STS timestamps
        Args:
            csv_file: Path to input CSV file
            time_scale: Time scaling factor (0.001 = 1000x faster than real-time)
        """
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        df['sts'] = pd.to_datetime(df['sts'])

        # Sort by STS (server timestamp - when data arrives)
        df = df.sort_values('sts').reset_index(drop=True)

        print(f"Total events to process: {len(df)}")
        print(f"Time range: {df['sts'].min()} to {df['sts'].max()}")
        print(f"Unique devices: {df['device_fk_id'].nunique()}")
        print("\nStarting stream processing...\n")
        start_sts = df['sts'].iloc[0]
        start_real_time = datetime.now()

        events_processed = 0

        for idx, row in df.iterrows():
            time_offset = (row['sts'] - start_sts).total_seconds()
            scaled_offset = time_offset * time_scale

            target_time = start_real_time + timedelta(seconds=scaled_offset)
            current_time = datetime.now()

            if current_time < target_time:
                sleep_duration = (target_time - current_time).total_seconds()
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            # Process the event
            event = row.to_dict()
            current_time = datetime.now()
            updated = self.process_event(event, current_time)

            events_processed += 1

            # Progress update every 100 events
            if events_processed % 100 == 0:
                print(f"Processed {events_processed}/{len(df)} events...")

        print(f"\n✓ Stream processing complete! Processed {events_processed} events.")

    def save_output(self):
        """Save historical log to CSV"""
        if not self.history_log:
            print("No data to save.")
            return

        df = pd.DataFrame(self.history_log)
        df.to_csv(self.output_file, index=False)
        print(f"\n✓ Output saved to {self.output_file}")

        # Also save current latest state
        latest_state_file = self.output_file.replace('.csv', '_latest_state.csv')
        latest_df = pd.DataFrame([
            {
                'device_id': device_id,
                **state
            }
            for device_id, state in self.latest_state.items()
        ])
        latest_df.to_csv(latest_state_file, index=False)
        print(f"✓ Latest state saved to {latest_state_file}")

        # Duplicate events and geofence alerts (always written, so a file never holds stale results)
        duplicates_file = self.output_file.replace('.csv', '_duplicates.csv')
        pd.DataFrame(self.duplicate_log, columns=DUPLICATE_LOG_COLUMNS).to_csv(duplicates_file, index=False)
        print(f"✓ Duplicate events ({len(self.duplicate_log)}) saved to {duplicates_file}")

        alerts_file = self.output_file.replace('.csv', '_geofence_alerts.csv')
        pd.DataFrame(self.alert_log, columns=ALERT_LOG_COLUMNS).to_csv(alerts_file, index=False)
        print(f"✓ Geofence alerts ({len(self.alert_log)}) saved to {alerts_file}")

    def query_latest_at_time(self, device_id: str, query_time: datetime) -> Optional[dict]:
        """
        Query the latest location data for a device at a specific time
        Args:
            device_id: Device identifier
            query_time: Time to query (as datetime or string)
        Returns:
            Dictionary with latest location data at that time, or None if not found
        """
        if isinstance(query_time, str):
            query_time = pd.to_datetime(query_time)

        # Filter history for this device up to query_time
        device_history = [
            entry for entry in self.history_log
            if entry['device_id'] == str(device_id)
               and entry['processing_time'] <= query_time
        ]

        if not device_history:
            return None

        latest_entry = max(device_history, key=lambda x: x['event_timestamp'])

        return {
            'device_id': latest_entry['device_id'],
            'timestamp': latest_entry['event_timestamp'],
            'latitude': latest_entry['latitude'],
            'longitude': latest_entry['longitude'],
            'speed': latest_entry['speed'],
            'queried_at': query_time
        }

    def get_performance_metrics(self) -> dict:
        """Calculate and return performance metrics"""
        if not self.processing_times:
            return {}

        return {
            'total_events': self.total_events_processed,
            'average_processing_time_ms': np.mean(self.processing_times),
            'median_processing_time_ms': np.median(self.processing_times),
            'p95_processing_time_ms': np.percentile(self.processing_times, 95),
            'p99_processing_time_ms': np.percentile(self.processing_times, 99),
            'max_processing_time_ms': np.max(self.processing_times),
            'total_devices': len(self.latest_state),
            'duplicate_events': self.duplicate_events_count,
            'duplicate_events_by_device': dict(self.duplicate_count_by_device),
            'total_geofences': len(self.geofences),
            'geofence_entry_alerts': sum(1 for a in self.alert_log if a['alert_type'] == 'ENTRY'),
            'geofence_exit_alerts': sum(1 for a in self.alert_log if a['alert_type'] == 'EXIT')
        }

    def print_performance_report(self):
        """Print detailed performance report"""
        metrics = self.get_performance_metrics()

        print("\n" + "=" * 60)
        print("PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Total Events Processed: {metrics['total_events']}")
        print(f"Unique Devices: {metrics['total_devices']}")
        print(f"\nProcessing Time Statistics:")
        print(f"  Average: {metrics['average_processing_time_ms']:.4f} ms")
        print(f"  Median:  {metrics['median_processing_time_ms']:.4f} ms")
        print(f"  P95:     {metrics['p95_processing_time_ms']:.4f} ms")
        print(f"  P99:     {metrics['p99_processing_time_ms']:.4f} ms")
        print(f"  Max:     {metrics['max_processing_time_ms']:.4f} ms")
        print(f"\nDuplicate Events Ignored: {metrics['duplicate_events']}")
        for device_id, count in metrics['duplicate_events_by_device'].items():
            print(f"  Device {device_id}: {count}")
        print(f"\nGeofences Defined: {metrics['total_geofences']}")
        print(f"  Entry alerts: {metrics['geofence_entry_alerts']}")
        print(f"  Exit alerts:  {metrics['geofence_exit_alerts']}")
        print("=" * 60)
