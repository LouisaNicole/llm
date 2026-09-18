#!/bin/bash
string="hello world"
substring="ell"

if [[ "$string" == *"$substring"* ]]; then
    echo "字符串包含 '$substring'"
else
    echo "字符串不包含 '$substring'"
fi
