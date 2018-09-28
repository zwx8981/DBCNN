function encoder_save(encoder, filePath)

if isa(encoder.net, 'dagnn.DagNN')
    device = encoder.net.device;
else
    if encoder.net.useGpu
        device = 'gpu';
    else
        device = 'cpu';
    end
end

if isfield(encoder, 'net')
    encoder.net = net_move_to_device(encoder.net, 'cpu');
end

save(filePath, '-struct', 'encoder') ;

encoder.net = net_move_to_device(encoder.net, device);