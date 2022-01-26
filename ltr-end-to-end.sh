usage()
{
  echo "Usage: $0 [-s /workspace/search_with_machine_learning_course/src/main/python/search_ml] [-c {ctr, heuristic, binary}] [ -w week2 ] [ -a /path/to/bbuy/products/logstash/conf ] [ -e /path/to/bbuy/queries/logstash/conf ] [ -l /path/to/logstash/home ] [-r num rows to split train/test on, default 1000] [-y]"
  exit 2
}

SOURCE_DIR="/workspace/search_with_machine_learning_course/src/main/python/search_ml"
WEEK="week2"
OUTPUT_DIR="/workspace/ltr_output"
ALL_CLICKS_FILE="/workspace/datasets/bbuy/train.csv"
NUM_ROWS=1000
CLICK_MODEL="heuristic"
SYNTHESIZE=""
while getopts ':s:c:w:o:a:r:yh' c
do
  case $c in
    a) ALL_CLICKS_FILE=$OPTARG ;;
    c) CLICK_MODEL=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    r) NUM_ROWS=$OPTARG ;;
    s) SOURCE_DIR=$OPTARG ;;
    w) WEEK=$OPTARG ;;
    y) SYNTHESIZE="--synthesize" ;;
    h) usage ;;
    [?]) usage ;;
  esac
done
shift $((OPTIND -1))


# NOTE: You must run this with the appropriate Pyenv VENV active.
# $SOURCE_DIR -- path to base of source code, eg. search_with_machine_learning/src/main/python
# $WEEK -- what week to look into
# $OUTPUT_DIR -- directory where to write output to
# $ALL_CLICKS_FILE -- The set of all clicks
cd $SOURCE_DIR
mkdir -p $OUTPUT_DIR
echo "ltr-end-to-end.sh $SOURCE_DIR $WEEK $OUTPUT_DIR $ALL_CLICKS_FILE $SYNTHESIZE $CLICK_MODEL run at "  `date` > $OUTPUT_DIR/meta.txt
set -x
python $WEEK/utilities/build_ltr.py --create_ltr_store
if [ $? -ne 0 ] ; then
  exit 2
fi
python $WEEK/utilities/build_ltr.py -f $WEEK/conf/ltr_featureset.json --upload_featureset
if [ $? -ne 0 ] ; then
  exit 2
fi
# Create our impressios impressions (positive/negative) data set, e.g. all sessions (with LTR features added in already)
echo "Creating impressions data set"
#python $WEEK/utilities/build_ltr.py --generate_impressions "$OUTPUT_DIR" -g --impressions_file "$OUTPUT_DIR/impressions.csv" --query_ids "$OUTPUT_DIR/query_id_map.json" --all_clicks "$ALL_CLICKS_FILE"
python $WEEK/utilities/build_ltr.py --generate_impressions  --output_dir "$OUTPUT_DIR" --all_clicks "$ALL_CLICKS_FILE" $SYNTHESIZE
if [ $? -ne 0 ] ; then
  exit 2
fi
echo "Creating training and test data sets from impressions"
# Split the impressions into training and test
python $WEEK/utilities/build_ltr.py --output_dir "$OUTPUT_DIR" --split_input "$OUTPUT_DIR/impressions.csv"  --split_rows $NUM_ROWS
if [ $? -ne 0 ] ; then
  exit 2
fi
# Create the actual training set from the impressions set
python $WEEK/utilities/build_ltr.py --ltr_terms_field sku --output_dir "$OUTPUT_DIR" --create_xgb_training -f $WEEK/conf/ltr_featureset.json --click_model $CLICK_MODEL
if [ $? -ne 0 ] ; then
  exit 2
fi
#exit
#python $WEEK/utilities/build_ltr.py --ltr_terms_field _id --create_training --train_file "$OUTPUT_DIR/train.csv" --query_ids "$OUTPUT_DIR/query_id_map.json" -f $WEEK/conf/ltr_featureset.json --xgb_feat_map "$OUTPUT_DIR/xgb_feat_map.txt" --xgb_train_output "$OUTPUT_DIR/training.xgb"
# Given a training set in SVMRank format, train an XGB model
python $WEEK/utilities/build_ltr.py  --output_dir "$OUTPUT_DIR" -x "$OUTPUT_DIR/training.xgb" --xgb_conf $WEEK/conf/xgb-conf.json
if [ $? -ne 0 ] ; then
  exit 2
fi
# Given an XGB model, upload it to the LTR store
python $WEEK/utilities/build_ltr.py --upload_ltr_model --xgb_model "$OUTPUT_DIR/xgb_model.model"
if [ $? -ne 0 ] ; then
  exit 2
fi
# Dump out some useful plots for visualizing our model
python $WEEK/utilities/build_ltr.py --xgb_plot --output_dir "$OUTPUT_DIR"
if [ $? -ne 0 ] ; then
  exit 2
fi
# Run our test queries through
python $WEEK/utilities/build_ltr.py --xgb_test "$OUTPUT_DIR/test.csv" --all_clicks "$ALL_CLICKS_FILE"