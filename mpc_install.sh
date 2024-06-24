# download MP-SPDZ
wget https://github.com/data61/MP-SPDZ/releases/download/v0.3.8/mp-spdz-0.3.8.tar.xz

tar -xf mp-spdz-0.3.8.tar.xz
rm -r mp-spdz-0.3.8.tar.xz

rsync -av ./mpspdz/ ./mp-spdz-0.3.8/

rm -r mpspdz

mv mp-spdz-0.3.8 mpspdz


# setup MP-SPDZ
cd mpspdz/Scripts/
chmod +x tldr.sh
./tldr.sh