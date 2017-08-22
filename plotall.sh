for i in 0 1 2 3
do
{
  if [ $i -eq 0 ]
  then
    python testbed.py
  elif [ $i -eq 1 ]
  then
    python testbed.py --acc
  elif [ $i -eq 2 ]
  then
    python testbed.py --pos
  elif [ $i -eq 3 ]
  then
    python testbed.py --acc --pos
  fi
}&
done
wait
