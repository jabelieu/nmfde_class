#!bin/bash

message="Felt cute, might revert later."

git add -A

git commit -m "$message"

git push origin main

echo 
echo "=================="
echo " Pushed to GitHub "
echo "=================="
echo