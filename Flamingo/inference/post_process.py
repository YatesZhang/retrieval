import re 


def remove_special_characters(text):
    # 匹配 ASCII 特殊字符的正则表达式
    pattern = r'[^\x00-\x7f]+'
    
    # 使用空字符串替换匹配到的特殊字符
    result = re.sub(pattern, ' ', text)
    return result

def post_process(texts, cats=None):
    """ 
        post process for GTSRB dataset
    """
    result = []
    for text in texts:
        skip_flag = False
        text = text.replace("\n", " ")
        text = remove_special_characters(text)
        if 'No passing veh over 3.5 tons' in text:
            result.append('No passing veh over 3.5 tons')    # No passing 
            continue
        if cats is not None:
            for cat in cats:
                if cat in text:
                    result.append(cat)  
                    skip_flag = True
                    # print("skip: ", cat)
                    break 
        if skip_flag:
            continue 
        # text = text.replace("\n", "")
        text = re.sub(r'guiActive.*', '', text) 
        text = re.sub(r'\\.*', '', text) 
        result.append(text)
         
    return result