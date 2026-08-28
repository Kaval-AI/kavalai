#!/bin/bash
set -e

# Function to run backoffice migrations
run_backoffice_migrations() {
    echo "Running backoffice migrations..."
    # Reads KAVALAI_BO_DB_URI and KAVALAI_BO_DB_SCHEMA from the environment.
    python -m kavalai.migrate_db backoffice
}

# Function to run backoffice server
run_backoffice_server() {
    echo "Starting Nginx..."
    service nginx start

    echo "Starting backoffice server..."
    # KAVALAI_BO_FRONTEND_URL, the OAuth client and the session key come from the environment.
    exec python -m kavalai.backoffice.server
}

case "$1" in
    backoffice-migrations)
        run_backoffice_migrations
        ;;
    backoffice-server)
        run_backoffice_server
        ;;
    *)
        echo "Usage: $0 {backoffice-migrations|backoffice-server}"
        exit 1
        ;;
esac
