"""Module rieng cho test 2: gap vat bang PhysBrain + servo anh.

Cac module o day cam thang vao khe Source / Policy / guard / Sink cua
``SDK/pipeline``. Khong co harness rieng, khong doc lap voi SDK.

Khong re-export o day: ``guard`` va ``policy`` can ``pipeline`` tren sys.path,
nen import eager se khien ``vla.physbrain_model`` (thuan HTTP, khong dinh gi toi
SDK) cung khong import duoc. Ben goi import dung module minh can.
"""
