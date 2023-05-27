import gradio as gr

from search import Searcher


searcher = Searcher()


def query_search(query, top_k):
    """
    搜索最相似的图片
    """
    top_k = int(top_k)
    distance_list, index_list = searcher.search(query)
    # 查看 query 是否存在
    gold_ids = None
    if query in searcher.query2image_ids:
        gold_ids = searcher.query2image_ids[query]
    return searcher.display_10_images(query, index_list, gold_ids, top_k)


inputs = [
    gr.inputs.Textbox(lines=1, label="输入query"),
    gr.inputs.Slider(minimum=1, maximum=20, step=1, default=10, label="返回的图片数量"),
]
outputs = [
    gr.outputs.Image(type="pil", label="最相似的图片"),
]


demo = gr.Interface(
    fn=query_search,
    inputs=inputs,
    outputs=outputs,
    title="CLIP 检索图片",
)

demo.launch()
