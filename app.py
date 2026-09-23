import pandas as pd
import streamlit as st

def load_and_preprocess_data(uploaded_file):
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # 1. Clean column names: remove whitespace, lowercase, strip brackets like <DATE> -> date
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.lower()
            .str.replace('<', '', regex=False)
            .str.replace('>', '', regex=False)
        )
        
        # 2. Identify potential Date and Time columns
        date_col = None
        time_col = None
        datetime_col = None
        
        for col in df.columns:
            if col in ['date', 'datetime', 'timestamp', 'time']:
                if col in ['date', 'datetime', 'timestamp'] and not date_col:
                    date_col = col
                elif col == 'time' and not time_col:
                    time_col = col

        # 3. Handle separate Date and Time columns (common in MetaTrader exports)
        if date_col and time_col and date_col != time_col:
            df['full_timestamp'] = pd.to_datetime(
                df[date_col].astype(str) + ' ' + df[time_col].astype(str), 
                errors='coerce'
            )
            datetime_col = 'full_timestamp'
        elif date_col:
            df['full_timestamp'] = pd.to_datetime(df[date_col], errors='coerce')
            datetime_col = 'full_timestamp'
        else:
            # Fallback: check if the first column can be parsed as dates
            first_col = df.columns[0]
            parsed = pd.to_datetime(df[first_col], errors='coerce')
            if parsed.notna().sum() > len(df) * 0.5: # If >50% parse successfully
                df['full_timestamp'] = parsed
                datetime_col = 'full_timestamp'
                
        if not datetime_col or df['full_timestamp'].isna().all():
            st.error("Could not automatically find or parse a valid Date/Time column. Please check your CSV format.")
            return None
            
        # Set datetime as index or standard column
        df = df.dropna(subset=['full_timestamp'])
        df = df.sort_values('full_timestamp').reset_index(drop=True)
        
        return df

    except Exception as e:
        st.error(f"Error processing data or executing models: {str(e)}")
        return None
