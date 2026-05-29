with 

source as (

    select * from {{ source('flagship_weather', 'feat_weather_signals') }}

),

renamed as (

    select
        date,
        datatype,
        value,
        rolling_avg_7d,
        rolling_avg_30d,
        deviation,
        is_extreme,
        severity_score,
        consecutive_days

    from source

)

select * from renamed