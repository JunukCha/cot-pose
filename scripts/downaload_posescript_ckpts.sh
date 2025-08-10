wget https://download.europe.naverlabs.com/ComputerVision/PoseFix/capgen_CAtransfPSA2H2_dataPSA2ftPSH2.zip -P pretrained_models
unzip pretrained_models/capgen_CAtransfPSA2H2_dataPSA2ftPSH2.zip -d pretrained_models/pose2text/
rm pretrained_models/capgen_CAtransfPSA2H2_dataPSA2ftPSH2.zip

wget https://download.europe.naverlabs.com/ComputerVision/PoseFix/ret_distilbert_dataPSA2ftPSH2.zip -P pretrained_models
unzip pretrained_models/ret_distilbert_dataPSA2ftPSH2.zip -d pretrained_models/retrieval/
rm pretrained_models/ret_distilbert_dataPSA2ftPSH2.zip

wget https://download.europe.naverlabs.com/ComputerVision/PoseFix/gen_distilbert_dataPSA2ftPSH2.zip -P pretrained_models
unzip pretrained_models/gen_distilbert_dataPSA2ftPSH2.zip -d pretrained_models/text2pose/
rm pretrained_models/gen_distilbert_dataPSA2ftPSH2.zip

huggingface-cli download \
  distilbert-base-uncased \
  --local-dir posescript/tools/huggingface_models/distilbert-base-uncased