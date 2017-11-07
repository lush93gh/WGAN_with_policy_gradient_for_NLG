此實驗之程式碼使用tensorflow 1.0與python 2.7來架構與實作相關模型。
由於tensorflow不支援上下相容，因此使用非tensorflow 1.0的版本可能會
無法執行。程式碼預設使用GPU來執行，若無GPU則可將參數"USING_CPU"設置
成"True"(USING_CPU = True)便可使用CPU來運行代碼。另外，相關量化數值，
可進一步開啟tensorboard查閱之。謝謝。

SeqGAN 原作代碼:
https://github.com/LantaoYu/SeqGAN

