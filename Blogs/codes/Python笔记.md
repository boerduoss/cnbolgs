> [!note] 使用Python的好习惯
> - @datacalss
> - pathlib函数操作路径
> - logging取代print

### 使用 @dataclass 装饰器
对于简单的数据容器可以直接使用 `@dataclass` 装饰器，这样省去了手动写 `__init__`的麻烦，另外 `@dataclas` 装饰器还会自动实现 `__repr__` 和 `__eq__` 方法。

### 使用 pathlib 处理文件路径

旧方法采用 `os` 模块来处理文件路径，例如我们需要创建一个 `data/files/example.csv` 文件路径

```python
import os 

path_way = os.path.join("data", "files", "example.csv")
```

采用 `pathlib` 的实现方法如下，可以直接实例化 `path`

```python
from pathlib import Path

path_way = Path("data") / "files" / "example.csv"
```

当然这里看不出什么差别，关键其实在于 `pathlib` 和`os`设计理念的差别，`pathlib` 是更加现代的“面向对象”的程序接口，所以理所应当实例化对象`path_way`也可以调用类的方法，访问属性，例如：
```python
path_way.parent  #获取父目录
path_way.name    #获取文件名字
path_way.suffix  #文件后缀

path_way.write_text("hello, 0, 1, 1")  #写入文本
path_way.read_text("hello, 0, 1, 1")   #读取文本 
```

### 使用 `logging` 而不是 `print`
