import nest_asyncio
nest_asyncio.apply()

from .ui.app import main

if __name__ == '__main__':
    main()