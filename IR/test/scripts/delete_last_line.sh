# retrieve last line from file
FILE=$1
LAST=$(tail -n 1 $FILE)

# truncate file
let TRUNCATE_SIZE="${#LAST} + 1"
truncate -s -"$TRUNCATE_SIZE" "$FILE"

# ... $LAST contains 'popped' last line

