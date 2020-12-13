# ./bootstrap_full_res/**/*.jpg;
for f in ./bootstrap_full_res/*.jpg; do
    FP=$(basename "$f")
    echo "$FP"
    convert "$f" -crop 30x30! -resize 30x30! -colorspace sRGB -type truecolor "./images/%d_$FP"
done

python -c 'import os, json; print json.dumps(os.listdir("./images"))' > images.json