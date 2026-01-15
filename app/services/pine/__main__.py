"""Enable running as: python -m app.services.pine --build ..."""

from app.services.pine.registry import main

if __name__ == "__main__":
    raise SystemExit(main())
