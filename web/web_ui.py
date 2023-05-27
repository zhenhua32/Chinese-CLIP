import gradio as gr

from search import Searcher


searcher = Searcher()


def query_search(query, top_k):
    """
    搜索最相似的图片
    """
    top_k = int(top_k)
    distance_list, index_list = searcher.search(query, top_k)
    # 查看 query 是否存在
    gold_ids = None
    if query in searcher.query2image_ids:
        gold_ids = searcher.query2image_ids[query]
    return searcher.display_images(query, index_list, gold_ids, top_k)


inputs = [
    gr.Textbox(lines=1, label="输入query"),
    gr.Slider(minimum=1, maximum=20, step=1, default=10, label="返回的图片数量"),
]
outputs = [
    gr.Image(type="pil", label="最相似的图片"),
]

# 我还是想要直接瀑布流的
with gr.Blocks() as demo:
    gr.Markdown("# CLIP 检索图片")
    gr.Textbox
    query_input = gr.Textbox(lines=1, label="输入query")
    top_k_input = gr.Slider(minimum=1, maximum=20, step=1, value=10, label="返回的图片数量")
    search_button = gr.Button(value="搜索")
    image_output = gr.Image(type="pil", label="最相似的图片")

    # 绑定搜索按钮
    search_button.click(
        query_search,
        inputs=[query_input, top_k_input],
        outputs=[image_output],
    )


demo.launch()
