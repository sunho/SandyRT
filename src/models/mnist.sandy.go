func layer1(x Tensor) Tensor {
    weight_scope "fc1" {
        x = __linear(x, @weight, @bias)
        x = __relu(x)
        return x
    }
}

func layer2(x Tensor) Tensor {
    weight_scope "fc2" {
        x = __linear(x, @weight, @bias)
        return x
    }
}

func main(x Tensor) Tensor {
    x = layer1(x)
    x = layer2(x)
    return x
}
