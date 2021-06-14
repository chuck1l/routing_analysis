import pandas as pd
import numpy as np


def flr_query(state, con):

    rr_sql = f'''
        SELECT 
            flr.ridecreate_datetime, flr.date_ride, flr.ride_status,
            flr.ridecreate_user_code, flr.call_center_state,
            flr.advanced_notice, flr.treatment_type_group,
            flr.los_code, flr.rider_age, flr.ride_id,
            flr.contract_status, flr.provider_type_name, flr.provider_status,
            flr.pu_city_code, flr.pu_county_code, flr.pu_zip_code,
            flr.miles, flr.cost_usd, flr.do_county_density,
            flr.first_updated_by, flr.last_updated_by, flr.midpoint_target_market,
            flr.provider_code, flr.prm, flr.override_cost_usd, flr.total_passengers,
            CASE WHEN flr.is_lyft = TRUE THEN 1 ELSE 0 END AS is_lyft,
            CASE WHEN flr.reroute_within_24_hrs = 'Y' THEN 1 ELSE 0 END AS reroute_24hrs
        FROM dw.fact_lcad_ride flr
        WHERE pu_state = '{state}'
            AND flr.provider_type_name IN ('Taxi Non-contracted', 'Non-Dedicated', 'Dedicated')
            AND flr.advanced_notice NOT IN ('Urgent', '1 Day', 'Same Day')
            AND flr.los_desc IN ('Ambulatory', 'Wheelchair')
            AND flr.date_ride BETWEEN dateadd('day', -120, current_date) and dateadd('day', -0, current_date)
            AND flr.provider_is_lcad = 'LCAD'
            AND flr.ride_status != 'Cancelled'
            AND flr.ridecreate_datetime IS NOT NULL
            AND flr.date_ride IS NOT NULL;'''

    my_data = pd.read_sql_query(rr_sql, con)

    my_data.to_csv(f'../data/reroute_full_data_{state}.csv')
    print(f'The {state} data are collected.')
    
    return None #my_data


if __name__ == '__main__':
   pass
