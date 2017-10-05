echo "Lenta:"
python3 evaluate.py --lemma gold/Lenta_gold.txt tagged/Lenta_extracted.txt
echo "VK:"
python3 evaluate.py --lemma gold/VK_gold.txt tagged/VK_extracted.txt
echo "JZ:"
python3 evaluate.py --lemma gold/JZ_gold.txt tagged/JZ_extracted.txt
