function net = net_move_to_device(net, device)

isDag = isa(net, 'dagnn.DagNN');

if isDag
   net.move(device) 
else
   net = vl_simplenn_move(net, device) ;
   if strcmp(device, 'gpu')
       net.useGpu = true;
   else
       net.useGpu = false;
   end
end