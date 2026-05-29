with weather_signals as (
    select *
    from {{ ref('stg_weather_signals') }}
),

dates as (
    select *
    from {{ ref('stg_date') }}
),

final as (
    select
        w.rolling_avg_7d,
        w.rolling_avg_30d,
        w.is_extreme,
        w.severity_score,
        w.consecutive_days,
        w.deviation,
        w.datatype,
        w.value,
        d.date,
        d.year,
        d.month,
        d.day,
        d.season,
        d.is_weekend
    from weather_signals as w
    left join dates d
        on w.date = d.date
)

select * from final