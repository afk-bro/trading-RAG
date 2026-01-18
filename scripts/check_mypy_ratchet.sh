#!/bin/bash
# Mypy ratchet: fail if ignore count grows beyond baseline
# Update BASELINE when you intentionally fix modules (decrease is good!)

BASELINE=62
CURRENT=$(grep -c "ignore_errors = True" mypy.ini 2>/dev/null || echo 0)

if [ "$CURRENT" -gt "$BASELINE" ]; then
    echo "ERROR: mypy.ini ignore count increased ($BASELINE -> $CURRENT)"
    echo "New ignores require justification. Fix types or update BASELINE if justified."
    exit 1
elif [ "$CURRENT" -lt "$BASELINE" ]; then
    echo "INFO: mypy.ini ignore count decreased ($BASELINE -> $CURRENT) - consider updating BASELINE"
fi

echo "OK: mypy ignore count ($CURRENT) <= baseline ($BASELINE)"
exit 0
