// ============================================================================
// dbwfb2_linebuf : one image-row delay line (W x 8-bit), asynchronous read.
//   dout = mem[addr]            (combinational: the value written one row ago)
//   mem[addr] <= din  when we   (synchronous write, same address, same cycle)
// Eight of these are cascaded in dbwfb2_ll_engine to give the 9-row vertical window.
// ============================================================================
module dbwfb2_linebuf #(
    parameter W  = 256,
    parameter AW = 8
)(
    input  wire                clk,
    input  wire                we,
    input  wire [AW-1:0]       addr,
    input  wire signed [7:0]   din,
    output wire signed [7:0]   dout
);
    (* ram_style = "distributed" *) reg signed [7:0] mem [0:W-1];

    integer i;
    initial begin
        for (i = 0; i < W; i = i + 1) mem[i] = 8'sd0;
    end

    assign dout = mem[addr];

    always @(posedge clk) begin
        if (we) mem[addr] <= din;
    end
endmodule