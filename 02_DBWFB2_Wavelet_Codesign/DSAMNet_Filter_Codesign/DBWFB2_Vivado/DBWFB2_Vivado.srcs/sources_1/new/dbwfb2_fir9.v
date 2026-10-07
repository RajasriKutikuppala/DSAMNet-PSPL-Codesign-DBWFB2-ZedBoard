// ============================================================================
// dbwfb2_fir9 : multiplier-free 9-tap symmetric FIR, two pipeline stages.
//   taps (x1024): 27, -17, -78, 273, 614, 273, -78, -17, 27   (sum = 1024 -> DC gain 1)
//   y = ( sum_k C[k]*t[k] + 512 ) >>> 10       (round-half-up, arithmetic shift)
//   stage 1 registers the five shift-add products, stage 2 registers y.
//   Both stages advance only when 'en' is high (global stall of the engine).
//   Latency from taps to y: 2 enabled clock edges.   No DSP blocks are used.
// ============================================================================
module dbwfb2_fir9 #(
    parameter IW = 8,     // input tap width (signed)
    parameter OW = 10     // output width (signed, after >>>10)
)(
    input  wire                 clk,
    input  wire                 en,
    input  wire signed [IW-1:0] t0, t1, t2, t3, t4, t5, t6, t7, t8,
    output reg  signed [OW-1:0] y
);
    localparam AW = 24;   // accumulator width: |sum| < 2^19 for the ranges used here

    // symmetric pair sums (one extra bit)
    wire signed [IW:0] s08 = t0 + t8;
    wire signed [IW:0] s17 = t1 + t7;
    wire signed [IW:0] s26 = t2 + t6;
    wire signed [IW:0] s35 = t3 + t5;

    // sign-extend to accumulator width
    wire signed [AW-1:0] e08 = s08;
    wire signed [AW-1:0] e17 = s17;
    wire signed [AW-1:0] e26 = s26;
    wire signed [AW-1:0] e35 = s35;
    wire signed [AW-1:0] e4  = t4;

    // shift-add products
    //   27  = 16+8+2+1          17  = 16+1           78  = 64+8+4+2
    //   273 = 256+16+1          614 = 512+64+32+4+2
    wire signed [AW-1:0] m27  = (e08 <<< 4) + (e08 <<< 3) + (e08 <<< 1) + e08;
    wire signed [AW-1:0] m17  = (e17 <<< 4) + e17;
    wire signed [AW-1:0] m78  = (e26 <<< 6) + (e26 <<< 3) + (e26 <<< 2) + (e26 <<< 1);
    wire signed [AW-1:0] m273 = (e35 <<< 8) + (e35 <<< 4) + e35;
    wire signed [AW-1:0] m614 = (e4 <<< 9) + (e4 <<< 6) + (e4 <<< 5) + (e4 <<< 2) + (e4 <<< 1);

    reg signed [AW-1:0] r27, r17, r78, r273, r614;
    always @(posedge clk) begin
        if (en) begin
            r27  <= m27;
            r17  <= m17;
            r78  <= m78;
            r273 <= m273;
            r614 <= m614;
        end
    end

    wire signed [AW-1:0] total = r614 + r273 - r78 - r17 + r27 + 512;

    always @(posedge clk) begin
        if (en) y <= total >>> 10;
    end
endmodule