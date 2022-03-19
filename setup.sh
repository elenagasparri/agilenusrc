#!/bin/bash


SETUP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#
# Base package root. All the other releavant folders are relative to this
# location.
#
export AGILENUSRC_ROOT=$SETUP_DIR
echo "AGILENUSRC_ROOT set to " $AGILENUSRC_ROOT

#
# Add the root folder to the $PYTHONPATH so that we can effectively import
# the relevant modules.
#
export PYTHONPATH=$AGILENUSRC_ROOT:$PYTHONPATH
echo "PYTHONPATH set to " $PYTHONPATH
