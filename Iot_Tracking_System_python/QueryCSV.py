import pandas as pd
from typing import Optional
from datetime import datetime

def query_from_csv(output_csv: str, device_id: str, query_time: datetime) -> Optional[dict]:
    """
    Standalone function to query location data from output CSV
    Args:
        output_csv: Path to output CSV file
        device_id: Device identifier
        query_time: Time to query (ISO format string)
    Returns:
        Dictionary with latest location data, or None if not found
    """
    df = pd.DataFrame(pd.read_csv(output_csv))
    df['processing_time'] = pd.to_datetime(df['processing_time'])
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    query_time_dt = pd.to_datetime(query_time)

    filtered = df[
        (df['device_id'] == str(device_id)) &
        (df['processing_time'] <= query_time_dt)
        ]

    if filtered.empty:
        return None

    latest = filtered.loc[filtered['event_timestamp'].idxmax()]
    return {
        'device_id': latest['device_id'],
        'timestamp': str(latest['event_timestamp']),
        'latitude': latest['latitude'],
        'longitude': latest['longitude'],
        'speed': latest['speed'],
        'queried_at': query_time
    }
