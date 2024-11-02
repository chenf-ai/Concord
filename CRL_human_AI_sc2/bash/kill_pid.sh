#!/bin/bash
ps -ef|grep -v grep|grep "StarCraftII/Versions"|awk '{print $2}'|while read pid
do
kill -9  $pid
done