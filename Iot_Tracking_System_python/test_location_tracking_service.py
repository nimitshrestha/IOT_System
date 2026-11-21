"""
Tests for duplicate event detection and geofencing in IoTLocationService.
Run with:  python -m unittest test_location_tracking_service -v
"""
import contextlib
import io
import logging
import math
import os
import random
import tempfile
import threading
import unittest
from datetime import datetime

import pandas as pd

from Location_tracking_service import (
    ALERT_LOG_COLUMNS, DUPLICATE_LOG_COLUMNS, EARTH_RADIUS_M, Geofence, IoTLocationService,
    haversine_distance,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NOW = datetime(2025, 1, 1, 12, 0, 0)

# Geofence centre used by most tests
CENTER_LAT, CENTER_LON = 27.8779, 76.0610


def setUpModule():
    # Keep test output clean; assertLogs still captures records from this logger.
    service_logger = logging.getLogger('Location_tracking_service')
    service_logger.addHandler(logging.NullHandler())
    service_logger.propagate = False


def north_of(lat, lon, metres):
    """Point `metres` due north of (lat, lon); along a meridian haversine distance is exactly R·Δφ."""
    return lat + math.degrees(metres / EARTH_RADIUS_M), lon


def event(device, ts, lat, lon, speed=0, sts=None):
    return {'device_fk_id': device, 'time_stamp': ts, 'latitude': lat, 'longitude': lon,
            'sts': sts or ts, 'speed': speed}


def ts(second):
    return f"2021-10-23T12:00:{second:02d}Z"


class HaversineTests(unittest.TestCase):
    def test_same_point_is_zero(self):
        self.assertEqual(haversine_distance(27.87, 76.06, 27.87, 76.06), 0.0)

    def test_one_degree_along_equator_and_meridian(self):
        one_degree = 2 * math.pi * EARTH_RADIUS_M / 360  # 111,194.93 m
        self.assertAlmostEqual(haversine_distance(0, 0, 0, 1), one_degree, places=4)
        self.assertAlmostEqual(haversine_distance(10, 20, 11, 20), one_degree, places=4)

    def test_antipodal_points(self):
        self.assertAlmostEqual(haversine_distance(0, 0, 0, 180), math.pi * EARTH_RADIUS_M, places=3)

    def test_symmetric_and_matches_spherical_law_of_cosines(self):
        rng = random.Random(42)
        for _ in range(200):
            lat1, lat2 = rng.uniform(-89, 89), rng.uniform(-89, 89)
            lon1, lon2 = rng.uniform(-180, 180), rng.uniform(-180, 180)
            d = haversine_distance(lat1, lon1, lat2, lon2)
            self.assertAlmostEqual(d, haversine_distance(lat2, lon2, lat1, lon1), places=6)
            p1, p2 = math.radians(lat1), math.radians(lat2)
            cos_c = (math.sin(p1) * math.sin(p2)
                     + math.cos(p1) * math.cos(p2) * math.cos(math.radians(lon2 - lon1)))
            expected = EARTH_RADIUS_M * math.acos(max(-1.0, min(1.0, cos_c)))
            self.assertAlmostEqual(d, expected, delta=1.0)


class GeofenceModelTests(unittest.TestCase):
    def setUp(self):
        self.gf = Geofence('GF', 'Hub', CENTER_LAT, CENTER_LON, 1000)

    def test_inside_outside_and_signed_boundary_distance(self):
        self.assertTrue(self.gf.contains(CENTER_LAT, CENTER_LON))
        self.assertAlmostEqual(self.gf.distance_to_boundary(CENTER_LAT, CENTER_LON), -1000, places=6)

        self.assertTrue(self.gf.contains(*north_of(CENTER_LAT, CENTER_LON, 999)))
        self.assertFalse(self.gf.contains(*north_of(CENTER_LAT, CENTER_LON, 1001)))
        self.assertAlmostEqual(self.gf.distance_to_boundary(*north_of(CENTER_LAT, CENTER_LON, 750)), -250, places=3)
        self.assertAlmostEqual(self.gf.distance_to_boundary(*north_of(CENTER_LAT, CENTER_LON, 3500)), 2500, places=3)

    def test_invalid_definitions_rejected(self):
        for lat, lon, radius in [(91, 0, 10), (-91, 0, 10), (0, 181, 10), (0, -181, 10),
                                 (0, 0, 0), (0, 0, -5), (0, 0, float('nan')), (0, 0, float('inf')),
                                 (float('nan'), 0, 10)]:
            with self.assertRaises(ValueError, msg=(lat, lon, radius)):
                Geofence('bad', 'Bad', lat, lon, radius)


class DuplicateDetectionTests(unittest.TestCase):
    def setUp(self):
        self.service = IoTLocationService()

    def test_duplicate_is_counted_logged_and_not_applied(self):
        self.assertTrue(self.service.process_event(event(25029, ts(0), 27.0, 76.0), NOW))
        with self.assertLogs('Location_tracking_service', level='WARNING') as logs:
            self.assertFalse(self.service.process_event(event(25029, ts(0), 27.0, 76.0, sts=ts(9)), NOW))

        self.assertIn('Duplicate event ignored: device=25029', logs.output[0])
        self.assertEqual(self.service.duplicate_events_count, 1)
        self.assertEqual(self.service.duplicate_count_by_device, {'25029': 1})
        self.assertEqual(len(self.service.duplicate_log), 1)
        self.assertEqual(self.service.duplicate_log[0]['sts'], pd.Timestamp(ts(9)))
        self.assertEqual(len(self.service.history_log), 1)         # duplicate kept out of history
        self.assertEqual(self.service.total_events_processed, 2)   # but it was received/processed

    def test_duplicate_with_conflicting_payload_does_not_update_latest_state(self):
        self.service.process_event(event('7', ts(5), 10.0, 20.0, speed=3), NOW)
        self.service.process_event(event('7', ts(5), 55.0, 66.0, speed=99), NOW)
        state = self.service.latest_state['7']
        self.assertEqual((state['latitude'], state['longitude'], state['speed']), (10.0, 20.0, 3))
        self.assertEqual(self.service.duplicate_events_count, 1)

    def test_key_is_normalised_across_id_and_timestamp_formats(self):
        self.service.process_event(event(25029, '2021-10-23T14:08:02Z', 1.0, 1.0), NOW)
        self.service.process_event(event('25029', pd.Timestamp('2021-10-23 14:08:02+00:00'), 1.0, 1.0), NOW)
        self.assertEqual(self.service.duplicate_events_count, 1)

    def test_same_timestamp_other_device_and_same_device_other_timestamp_are_not_duplicates(self):
        self.service.process_event(event('A', ts(1), 1.0, 1.0), NOW)
        self.service.process_event(event('B', ts(1), 1.0, 1.0), NOW)
        self.service.process_event(event('A', ts(2), 1.0, 1.0), NOW)
        self.service.process_event(event('A', ts(0), 1.0, 1.0), NOW)  # late but unique
        self.assertEqual(self.service.duplicate_events_count, 0)
        self.assertEqual(len(self.service.history_log), 4)

    def test_counters_per_device(self):
        for device, second in [('A', 1), ('A', 1), ('A', 1), ('B', 2), ('B', 2), ('C', 3)]:
            self.service.process_event(event(device, ts(second), 1.0, 1.0), NOW)
        self.assertEqual(self.service.duplicate_events_count, 3)
        self.assertEqual(self.service.duplicate_count_by_device, {'A': 2, 'B': 1})
        metrics = self.service.get_performance_metrics()
        self.assertEqual(metrics['duplicate_events'], 3)
        self.assertEqual(metrics['duplicate_events_by_device'], {'A': 2, 'B': 1})

    def test_concurrent_copies_accepted_exactly_once(self):
        threads_count = 16
        barrier = threading.Barrier(threads_count)
        results = []

        def submit():
            barrier.wait()
            results.append(self.service.process_event(event('X', ts(3), 1.0, 1.0), NOW))

        threads = [threading.Thread(target=submit) for _ in range(threads_count)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count(True), 1)
        self.assertEqual(self.service.duplicate_events_count, threads_count - 1)
        self.assertEqual(len(self.service.history_log), 1)


class GeofencingTests(unittest.TestCase):
    def setUp(self):
        self.service = IoTLocationService()
        self.service.add_geofence('GF-1', 'Hub', CENTER_LAT, CENTER_LON, 1000)
        self.inside = north_of(CENTER_LAT, CENTER_LON, 200)
        self.inside_2 = north_of(CENTER_LAT, CENTER_LON, 600)
        self.outside = north_of(CENTER_LAT, CENTER_LON, 1500)
        self.outside_2 = north_of(CENTER_LAT, CENTER_LON, 4000)

    def send(self, device, second, point):
        return self.service.process_event(event(device, ts(second), *point), NOW)

    def alerts(self):
        return [(a['device_id'], a['geofence_id'], a['alert_type']) for a in self.service.alert_log]

    def test_entry_then_exit_without_repeated_alerts(self):
        self.send('D1', 0, self.outside)      # baseline outside: no alert
        self.send('D1', 1, self.outside_2)    # still outside: no alert
        self.assertEqual(self.alerts(), [])

        self.send('D1', 2, self.inside)       # ENTRY
        self.send('D1', 3, self.inside_2)     # still inside
        self.send('D1', 4, self.inside)       # still inside
        self.assertEqual(self.alerts(), [('D1', 'GF-1', 'ENTRY')])

        self.send('D1', 5, self.outside)      # EXIT
        self.send('D1', 6, self.outside_2)    # still outside
        self.assertEqual(self.alerts(), [('D1', 'GF-1', 'ENTRY'), ('D1', 'GF-1', 'EXIT')])

        entry, exit_ = self.service.alert_log
        self.assertEqual(entry['event_timestamp'], pd.Timestamp(ts(2)))
        self.assertEqual(entry['previous_event_timestamp'], pd.Timestamp(ts(1)))
        self.assertEqual(entry['geofence_name'], 'Hub')
        self.assertAlmostEqual(entry['distance_to_boundary_m'], -800, places=3)
        self.assertAlmostEqual(exit_['distance_to_boundary_m'], 500, places=3)

    def test_first_location_inside_sets_baseline_without_alert(self):
        self.send('D1', 0, self.inside)
        self.assertEqual(self.alerts(), [])
        self.assertTrue(self.service.geofence_states[('D1', 'GF-1')]['inside'])
        self.send('D1', 1, self.outside)
        self.assertEqual(self.alerts(), [('D1', 'GF-1', 'EXIT')])

    def test_alerts_are_logged(self):
        self.send('D1', 0, self.outside)
        with self.assertLogs('Location_tracking_service', level='INFO') as logs:
            self.send('D1', 1, self.inside)
        self.assertTrue(any("GEOFENCE ENTRY: device D1 entered 'Hub' (GF-1)" in line for line in logs.output))

    def test_state_is_separate_per_device_geofence_pair(self):
        # GF-2 overlaps GF-1's centre; D1 and D2 move independently
        self.service.add_geofence('GF-2', 'Overlap', CENTER_LAT, CENTER_LON, 300)
        self.send('D1', 0, self.outside)
        self.send('D2', 0, self.inside)        # D2 baseline: inside GF-1 and GF-2
        self.send('D1', 1, self.inside_2)      # D1 enters GF-1 only (600 m > GF-2's 300 m)
        self.assertEqual(self.alerts(), [('D1', 'GF-1', 'ENTRY')])

        self.send('D1', 2, self.inside)        # D1 enters GF-2; GF-1 state unchanged
        self.send('D2', 1, self.inside_2)      # D2 exits GF-2 only
        self.assertEqual(self.alerts(), [('D1', 'GF-1', 'ENTRY'), ('D1', 'GF-2', 'ENTRY'),
                                         ('D2', 'GF-2', 'EXIT')])
        states = {k: v['inside'] for k, v in self.service.geofence_states.items()}
        self.assertEqual(states, {('D1', 'GF-1'): True, ('D1', 'GF-2'): True,
                                  ('D2', 'GF-1'): True, ('D2', 'GF-2'): False})

    def test_late_out_of_order_event_does_not_trigger_alert(self):
        self.send('D1', 0, self.outside)
        self.send('D1', 10, self.inside)                        # ENTRY
        self.assertFalse(self.send('D1', 5, self.outside))      # stale: where it *was*
        self.send('D1', 11, self.inside_2)
        self.assertEqual(self.alerts(), [('D1', 'GF-1', 'ENTRY')])
        self.assertTrue(self.service.geofence_states[('D1', 'GF-1')]['inside'])

    def test_duplicate_event_does_not_trigger_alert(self):
        self.send('D1', 0, self.outside)
        self.send('D1', 0, self.inside)                         # duplicate key, conflicting payload
        self.assertEqual(self.alerts(), [])
        self.assertFalse(self.service.geofence_states[('D1', 'GF-1')]['inside'])

    def test_geofence_added_mid_stream(self):
        self.send('D1', 0, self.inside)
        self.service.add_geofence('GF-NEW', 'New', CENTER_LAT, CENTER_LON, 100)
        self.send('D1', 1, north_of(CENTER_LAT, CENTER_LON, 50))   # baseline for GF-NEW, no alert
        self.send('D1', 2, north_of(CENTER_LAT, CENTER_LON, 150))  # exits GF-NEW only
        self.assertEqual(self.alerts(), [('D1', 'GF-NEW', 'EXIT')])

    def test_duplicate_geofence_id_rejected(self):
        with self.assertRaises(ValueError):
            self.service.add_geofence('GF-1', 'Again', 0, 0, 10)
        self.assertEqual(self.service.geofences['GF-1'].name, 'Hub')

    def test_device_geofence_status(self):
        self.assertEqual(self.service.get_device_geofence_status('unknown'), [])
        self.send('D1', 0, self.inside)
        self.service.add_geofence('GF-FAR', 'Far', 0.0, 0.0, 500)   # added after the last update
        status = {s['geofence_id']: s for s in self.service.get_device_geofence_status('D1')}
        self.assertTrue(status['GF-1']['inside'])
        self.assertAlmostEqual(status['GF-1']['distance_to_boundary_m'], -800, places=1)
        self.assertFalse(status['GF-FAR']['inside'])
        self.assertGreater(status['GF-FAR']['distance_to_boundary_m'], 0)
        self.assertEqual(status['GF-1']['as_of'], pd.Timestamp(ts(0)))


class OutputTests(unittest.TestCase):
    def test_save_output_writes_duplicate_and_alert_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = IoTLocationService(output_file=os.path.join(tmp, 'out.csv'))
            service.add_geofence('GF-1', 'Hub', CENTER_LAT, CENTER_LON, 1000)
            service.process_event(event('D1', ts(0), *north_of(CENTER_LAT, CENTER_LON, 2000)), NOW)
            service.process_event(event('D1', ts(0), *north_of(CENTER_LAT, CENTER_LON, 2000)), NOW)
            service.process_event(event('D1', ts(1), CENTER_LAT, CENTER_LON), NOW)
            with contextlib.redirect_stdout(io.StringIO()):
                service.save_output()

            duplicates = pd.read_csv(os.path.join(tmp, 'out_duplicates.csv'))
            alerts = pd.read_csv(os.path.join(tmp, 'out_geofence_alerts.csv'))
            self.assertEqual(list(duplicates.columns), DUPLICATE_LOG_COLUMNS)
            self.assertEqual(list(alerts.columns), ALERT_LOG_COLUMNS)
            self.assertEqual(len(duplicates), 1)
            self.assertEqual(alerts[['device_id', 'geofence_id', 'alert_type']].values.tolist(),
                             [['D1', 'GF-1', 'ENTRY']])
            # The in-memory records must not carry fields that the file layout drops
            self.assertEqual(set(service.duplicate_log[0]), set(DUPLICATE_LOG_COLUMNS))
            self.assertEqual(set(service.alert_log[0]), set(ALERT_LOG_COLUMNS))

    def test_empty_logs_still_write_header_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = IoTLocationService(output_file=os.path.join(tmp, 'out.csv'))
            service.process_event(event('D1', ts(0), 1.0, 1.0), NOW)
            with contextlib.redirect_stdout(io.StringIO()):
                service.save_output()
            self.assertEqual(list(pd.read_csv(os.path.join(tmp, 'out_duplicates.csv')).columns),
                             DUPLICATE_LOG_COLUMNS)
            self.assertTrue(pd.read_csv(os.path.join(tmp, 'out_geofence_alerts.csv')).empty)


@unittest.skipUnless(os.path.exists(os.path.join(HERE, 'raw_data.csv')), 'raw_data.csv not available')
class RawDataIntegrationTest(unittest.TestCase):
    """Full dataset through the real streaming path (time_scale=0 -> no sleeping)."""

    def test_raw_data_duplicates_and_alerts(self):
        service = IoTLocationService()
        service.add_geofence('GF-1', 'Origin Yard', 27.885506, 76.269203, 1000)
        service.add_geofence('GF-2', 'Highway Checkpoint', 27.850435, 76.194588, 1500)
        service.add_geofence('GF-3', 'Destination Hub', 27.877958, 76.060959, 1000)
        service.add_geofence('GF-4', 'Loading Bay', 19.739096, 76.181465, 400)
        service.add_geofence('GF-5', 'Warehouse', 21.615044, 78.436699, 200)
        with contextlib.redirect_stdout(io.StringIO()):
            service.stream_from_csv(os.path.join(HERE, 'raw_data.csv'), time_scale=0)

        raw = pd.read_csv(os.path.join(HERE, 'raw_data.csv'))
        expected_duplicates = int(raw.duplicated(subset=['device_fk_id', 'time_stamp']).sum())
        self.assertEqual(expected_duplicates, 6)
        self.assertEqual(service.duplicate_events_count, expected_duplicates)
        self.assertEqual(service.duplicate_count_by_device, {'25029': 6})
        self.assertEqual(service.total_events_processed, len(raw))
        self.assertEqual(len(service.history_log), len(raw) - expected_duplicates)

        # Expected alerts were derived independently (vectorised haversine replay of the
        # latest-state sequence); no repeated alerts while devices stay inside/outside.
        got = [(a['device_id'], a['geofence_id'], a['alert_type'], str(a['event_timestamp']))
               for a in service.alert_log]
        self.assertEqual(got, [
            ('25029', 'GF-1', 'EXIT', '2021-10-23 12:32:57+00:00'),
            ('25029', 'GF-2', 'ENTRY', '2021-10-23 13:07:04+00:00'),
            ('25029', 'GF-2', 'EXIT', '2021-10-23 13:08:30+00:00'),
            ('6888', 'GF-4', 'EXIT', '2021-10-23 13:12:48+00:00'),
            ('25029', 'GF-3', 'ENTRY', '2021-10-23 13:40:04+00:00'),
        ])
        self.assertEqual(len(service.geofence_states), 4 * 5)  # 4 devices x 5 geofences


if __name__ == '__main__':
    unittest.main()
