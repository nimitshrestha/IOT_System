from Location_tracking_service import IoTLocationService
from QueryCSV import query_from_csv
import json
import logging
from datetime import datetime

if __name__ == "__main__":
    # Duplicate events and geofence entry/exit alerts are reported through logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Initialize service
    service = IoTLocationService(output_file="location_output.csv")

    # Define geofences (id, name, centre latitude/longitude, radius in metres).
    # Example zones placed along the device routes in raw_data.csv.
    service.add_geofence("GF-1", "Origin Yard", 27.885506, 76.269203, 1000)        # 25029 starts inside, exits
    service.add_geofence("GF-2", "Highway Checkpoint", 27.850435, 76.194588, 1500)  # 25029 enters, then exits
    service.add_geofence("GF-3", "Destination Hub", 27.877958, 76.060959, 1000)     # 25029 enters and stays
    service.add_geofence("GF-4", "Loading Bay", 19.739096, 76.181465, 400)          # 6888 starts inside, exits
    service.add_geofence("GF-5", "Warehouse", 21.615044, 78.436699, 200)            # 20984 stays inside: no alerts

    # Process stream from CSV
    service.stream_from_csv("raw_data.csv", time_scale=0.001)

    # Save outputs
    service.save_output()

    # Print performance metrics
    service.print_performance_report()

    # Geofence alerts raised during the stream
    print("\n" + "=" * 60)
    print("GEOFENCE ALERTS")
    print("=" * 60)
    for alert in service.alert_log:
        print(f"{alert['alert_type']:<5} device {alert['device_id']:<6} {alert['geofence_name']:<20} "
              f"at {alert['event_timestamp']}  ({alert['distance_to_boundary_m']:+.1f} m from boundary)")

    # Current inside/outside status of every device for every geofence
    print("\n" + "=" * 60)
    print("CURRENT GEOFENCE STATUS")
    print("=" * 60)
    for device_id in service.latest_state:
        inside = [status['geofence_name'] for status in service.get_device_geofence_status(device_id) if status['inside']]
        print(f"Device {device_id}: inside {', '.join(inside) if inside else 'no geofence'}")

    # Example
    print("\n" + "=" * 60)
    print("EXAMPLE QUERIES")
    print("=" * 60)

    # Query using in-memory service
    result = service.query_latest_at_time('25029', datetime(2021, 10, 23, 14, 8, 10))
    if result:
        print(f"\nQuery 1 - Device 25029 at 2021-10-23T14:08:10Z:")
        print(json.dumps(result, indent=2, default=str))

    # Query using CSV file
    result2 = query_from_csv('location_output.csv', '25029', datetime(2021, 10, 23, 14, 8, 10))
    if result2:
        print(f"\nQuery 2 - From CSV file:")
        print(json.dumps(result2, indent=2, default=str))
