#bin/bash
set -e

wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
tar -xf montreal-forced-aligner_linux.tar.gz
rm montreal-forced-aligner_linux.tar.gz

pushd montreal-forced-aligner

# known mfa issue https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/issues/109
cp lib/libpython3.6m.so.1.0 lib/libpython3.6m.so

# download english dictionary from http://www.openslr.org/11/
wget https://www.openslr.org/resources/11/librispeech-lexicon.txt

# make sure executable is working
bin/mfa_align -h >/dev/null

popd

# cache folder mfa uses to store temporary alignments
mkdir -p ~/Documents/MFA
