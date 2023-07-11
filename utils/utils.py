
# 工具类
class Utils():
    def __init__(self) -> None:
        pass

    # 获取绘制图像标题
    def get_title(**kw):
        """
        将各个输入的各个参数生成字符串返回
        """
        _title = ''
        _index = 0
        for key, val in kw.items():
            _title += key + '=' + str(val)
            _index += 1
            if _index != len(kw.items()):
                _title += '\n'
        return _title







