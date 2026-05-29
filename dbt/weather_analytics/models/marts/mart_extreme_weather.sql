with observations as (
    select *
    from {{ ref('stg_noaa_observations') }}
    where is_extreme = true
),

dates as (
    select *
    from {{ ref('stg_date') }}
),

final as (
    select
        o.observation_id,
        o.datatype,
        o.value,
        d.date,
        d.year,
        d.month,
        d.day,
        d.season,
        d.is_weekend
    from observations o
    left join dates d
        on o.date_id = d.date_id
)

select * from final