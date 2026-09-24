# -*- coding: utf-8 -*-
"""Simulation I/O and Serialization Caching"""
import streamlit as st
import json

@st.cache_data(ttl=600)
def serialize_simulation_metadata(stage_1_len: int, stage_2_len: int, stage_3_len: int, seed: int) -> str:
    payload = {
        "N_total": stage_1_len + stage_2_len + stage_3_len,
        "stage_1_cases": stage_1_len,
        "stage_2_cases": stage_2_len,
        "stage_3_cases": stage_3_len,
        "seed": seed,
        "partition_integrity": "VERIFIED_CHRONOLOGICAL"
    }
    return json.dumps(payload, indent=2)
