
if [ ! -d "./PaddleDetection" ];then
	git clone --depth=1 https://github.com/PaddlePaddle/PaddleDetection.git
fi


sleep 2
cd PaddleDetection

if [ -d "./.git" ]; then
	sudo rm -rf .git
fi

if [ -e "./.gitignore" ];then
	sudo rm .gitignore
fi

sudo rm README*
