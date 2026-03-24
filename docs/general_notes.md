# Initial folder structure notes:

- /src change to include
    /src
      project_name/
        __init__.py
        main.py
        module1.py
        module2.py
- Adding __init__.py allows Python to treat it as a package, making importing modules cleaner and supports testing and future expansion

- Mirror src structure inside of tests
    - Keeps tests organized and makes it easy to scale when adding new modules

- For docs
    - Create a docs/README.md or docs/index.md to guide documentation structure
 
- Create a config/ folder to separate enviroment-specific settigns and keeps them out of src/
