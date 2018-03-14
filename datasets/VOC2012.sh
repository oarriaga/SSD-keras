#!/bin/bash
# Ellis Brown

start=`date +%s`

# handle optional download dir

echo "Downloading VOC2012 trainval ..."
# Download the data.
curl -LO http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
echo "Done downloading."


# Extract data
echo "Extracting trainval ..."
tar -xvf VOCtrainval_11-May-2012.tar
echo "removing tar ..."
rm VOCtrainval_11-May-2012.tar

end=`date +%s`
runtime=$((end-start))

echo "Completed in" $runtime "seconds"
