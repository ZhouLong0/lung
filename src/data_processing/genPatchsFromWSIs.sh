for slide_name in "19_D" "32_C" "133_A" "133_B" "192_A" "226_A" "226_C" "123_A"
do
    echo $slide_name
    python3 STEP0_generate_patchs.py --slide_name ${slide_name}
    echo "finished"
done