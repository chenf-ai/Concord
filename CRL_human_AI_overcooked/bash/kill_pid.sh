#!/bin/bash
ps -ef|grep -v grep|grep "offpg"|awk '{print $2}'|while read pid
do
kill -9  $pid
done