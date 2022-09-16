#!/bin/bash

while :
do
    mem_percent=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    now=$(date +'%H:%M:%S')
    echo "Time now: ${now} with ${mem_percent}% memory used"
    sleep 60 # a minute
done

