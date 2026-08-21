#!/bin/bash
set -e

# Function to run agent migrations
run_agent_migrations() {
    echo "Running agent migrations..."
    python -m kavalai.migrate_db app
}

# Function to run agent server
run_agent_server() {
    echo "Starting agent server..."
    if [ -z "$KAVALAI_AGENT_WORKFLOW_PATH" ]; then
        echo "Error: KAVALAI_AGENT_WORKFLOW_PATH environment variable is required to run agent server."
        exit 1
    fi
    # The entry point takes no arguments: it reads KAVALAI_AGENT_WORKFLOW_PATH,
    # KAVALAI_AGENT_HOST and KAVALAI_AGENT_PORT (plus the KAVALAI_DB_* settings)
    # from the environment itself.
    exec python -m kavalai.server
}

case "$1" in
    agent-migrations)
        run_agent_migrations
        ;;
    agent-server)
        run_agent_server
        ;;
    *)
        echo "Usage: $0 {agent-migrations|agent-server}"
        exit 1
        ;;
esac
