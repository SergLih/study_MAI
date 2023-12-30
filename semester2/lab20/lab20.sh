#!/bin/bash

ERROR1="Need exactly four arguments. File, number of copies, size and name of result file"
ERROR2="'$1' is not a file"
ERROR3="Please enter correct size 'Letters' or 'Figures'"


if (( $# == 0 )); then
    echo "Missing arguments"
    exit 1
else
    while(( "$#" )); do    
        COUNTER=0
        FILE=$1
        NUMBER_OF_COPIES=$2
        SIZE=$3
        NAME_FILE=$4
        
        if [[ $# -lt 4 ]]; then 
            echo $ERROR1 
            exit 1 
        fi 
    
        if [[ ! -f $FILE ]]; then 
            echo $ERROR2 
            exit 2 
        fi
        
        if [[ "$SIZE" != "Letters" ]] && [[ "$SIZE" != "Figures" ]]; then
            echo $ERROR3
            exit 3
        fi 

        if [[ "$SIZE" == "Letters" ]]; then
            if [[ $NAME_FILE =~ ^[a-z_]+$ ]]; then 
                while [[  $COUNTER -lt $2 ]]; do
                    cp $FILE $FILE$NAME_FILE
                    NAME_FILE=$(echo "$NAME_FILE" | tr "a-z" "b-z_")
                    if [[ "$NAME_FILE" == "_" ]]; then
                        FILE=$FILE"z"
                        NAME_FILE="a"
                    fi
                    let COUNTER=COUNTER+1
                done
                echo "Executed $2 copies source file, beginning with $4"
            else
                echo "Size has to be a word"
            fi
        fi

        if [[ "$SIZE" == "Figures" ]]; then
            if [[ $NAME_FILE =~ ^[0-9]+$ ]]; then   
                while [[  $COUNTER -lt $2 ]]; do
                    cp $FILE $FILE$NAME_FILE
                    let NAME_FILE=NAME_FILE+1
                    let COUNTER=COUNTER+1
                done
                echo "Executed $2 copies source file, beginning with $4"
            else
                echo "Size has to be a number"
            fi
        fi

        shift 4
    done
    exit 0
fi

