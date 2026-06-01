with 

source as (

    select * from {{ source('flagship_weather', 'dim_date') }}

),

renamed as (

    select
        date_id,
        date,
        year,
        month,
        day,
        season,
        is_weekend

    from source
    where date is not null

)

select * from renamed