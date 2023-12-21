# Mane-Project

*Models tested from:* https://www.sbert.net/docs/pretrained_models.html

*Demo Link:* https://www.canva.com/design/DAF25tpetPA/IegQltg6Ch4zd_sawJuqqA/edit


You need to clone the FastChat github on your local system, so that you can use vicuna inference : 

```bash
git clone https://github.com/lm-sys/FastChat.git
cd FastChat
```

If you are running on Mac:
```bash
brew install rust cmake
```

and then install these packages : 

```bash
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"
```


Make sure to install the packages in requirements.txt

```bash
pip install -r requirements.txt
```
