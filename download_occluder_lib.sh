wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DgBs8liddu0sf4PNcEKKkxjR-EqnOHY5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DgBs8liddu0sf4PNcEKKkxjR-EqnOHY5" -O OccludedLibs.zip && rm -rf /tmp/cookies.txt &&

unzip OccludedLibs.zip && 
mv OccludedLibs occluded_lib &&
rm OccludedLibs.zip && 