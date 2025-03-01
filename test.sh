rm -rf ./images/output
rm test-output.txt
rm test-output.csv
source ./venv/bin/activate
python test.py > test-output.txt