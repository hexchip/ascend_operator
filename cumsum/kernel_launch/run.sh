#!/bin/bash
rm -f cumsum
cp ./out/build/Debug_Ascend310B1/cumsum ./
rm -rf input output
mkdir -p input output
python3 scripts/gen_data.py
./cumsum
md5sum output/*.bin
python3 scripts/verify_result.py output/output.bin output/golden.bin
