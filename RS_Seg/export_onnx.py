from models.creat_model import create_model
import torch


def export_onnx():
    model = create_model(2, False)
    model.load_state_dict(torch.load(r'')["model"])
    model.eval()
    # Input to the model
    x = torch.randn(1, 4, 256, 256, requires_grad=True)
    torch_out = model(x)
    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      r"deep.onnx",  # where to save the model (can be a file or file-like object)
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})


if __name__ == '__main__':
    export_onnx()
