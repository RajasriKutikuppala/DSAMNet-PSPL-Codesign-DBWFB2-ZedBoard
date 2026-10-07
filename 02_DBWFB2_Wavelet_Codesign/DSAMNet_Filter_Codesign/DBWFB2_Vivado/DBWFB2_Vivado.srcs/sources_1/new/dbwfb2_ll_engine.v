// ============================================================================
// dbwfb2_ll_engine : streaming DBWFB2 LL-subband engine
//
//   in : AXI-Stream, int8, planar CHW raster, NP planes of H x W
//   out: AXI-Stream, int8, planar CHW raster, NP planes of H/2 x W/2
//
//   For every plane:
//     vertical pass   (8 line buffers, 9-row window, replicate-clamped at the top
//                      and bottom, keeps even rows)           V = (sum C*P + 512) >>> 10
//     horizontal pass (9-sample window, replicate-clamped at the left and right,
//                      keeps even columns)                    L = (sum C*V + 512) >>> 10
//     out = clip(L, -128, 127)
//   LL pixel (i,m) is centred on input pixel (2i,2m).  No state crosses a line,
//   a plane or an image boundary.  C = 27,-17,-78,273,614,273,-78,-17,27.
//
//   Control: one token per step travels with the data through the pipeline and
//   carries its own flags, so the output side needs no counters.  All registers are
//   gated by one global enable 'en' (stall when the output register holds a beat
//   that the receiver has not taken).
//   Throughput: one input byte per clock; (H+3)*W+3 steps per plane.
// ============================================================================
module dbwfb2_ll_engine #(
    parameter W  = 256,     // plane width  (even, >= 8)
    parameter H  = 256,     // plane height (even, >= 8)
    parameter NP = 3        // planes per image (3 = RGB)
)(
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axis:m_axis, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    input  wire [7:0]  s_axis_tdata,
    input  wire        s_axis_tvalid,
    output wire        s_axis_tready,

    output wire [7:0]  m_axis_tdata,
    output wire        m_axis_tvalid,
    input  wire        m_axis_tready,
    output wire        m_axis_tlast
);
    localparam AW = $clog2(W);
    localparam PW = (NP > 1) ? $clog2(NP) : 1;

    // ------------------------------------------------------------------ global enable
    reg  m_valid;                                   // output register state
    wire en = ~(m_valid & ~m_axis_tready);

    // ------------------------------------------------------------------ sequencer
    reg [PW-1:0]  p;            // plane
    reg [10:0]    r;            // row 0 .. H+3  (H..H+2 virtual rows, H+3 short flush row)
    reg [AW-1:0]  x;            // column

    wire in_row = (r < H);
    assign s_axis_tready = en & in_row;
    wire go = en & (in_row ? s_axis_tvalid : 1'b1);

    wire last_x = (r == H + 3) ? (x == 2) : (x == W - 1);

    always @(posedge aclk) begin
        if (!aresetn) begin
            p <= 0;
            r <= 0;
            x <= 0;
        end else if (go) begin
            if (last_x) begin
                x <= 0;
                if (r == H + 3) begin
                    r <= 0;
                    p <= (p == NP - 1) ? 0 : p + 1'b1;
                end else begin
                    r <= r + 1'b1;
                end
            end else begin
                x <= x + 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------ line buffers
    wire [63:0]       lbo;      // lbo[8*k +: 8] = output of line buffer k+1 = tap t(k+1)
    wire [63:0]       lbi;      // input of line buffer k+1
    wire signed [7:0] t1 = lbo[ 7: 0];
    wire signed [7:0] t2 = lbo[15: 8];
    wire signed [7:0] t3 = lbo[23:16];
    wire signed [7:0] t4 = lbo[31:24];
    wire signed [7:0] t5 = lbo[39:32];
    wire signed [7:0] t6 = lbo[47:40];
    wire signed [7:0] t7 = lbo[55:48];
    wire signed [7:0] t8 = lbo[63:56];
    // newest tap: the incoming pixel; in the virtual rows the last row is replicated
    wire signed [7:0] t0 = in_row ? $signed(s_axis_tdata) : t1;

    assign lbi = {lbo[55:0], t0};

    genvar gi;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : g_lb
            dbwfb2_linebuf #(.W(W), .AW(AW)) u_lb (
                .clk  (aclk),
                .we   (go),
                .addr (x),
                .din  (lbi[8*gi +: 8]),
                .dout (lbo[8*gi +: 8])
            );
        end
    endgenerate

    // top-of-image clamp: rows above row 0 repeat row 0
    reg signed [7:0] c0, c1, c2, c3, c4, c5, c6, c7, c8;
    always @* begin
        c0 = t0; c1 = t1; c2 = t2; c3 = t3; c4 = t4;
        c5 = t5; c6 = t6; c7 = t7; c8 = t8;
        if (r == 11'd4) begin
            c5 = t4; c6 = t4; c7 = t4; c8 = t4;
        end else if (r == 11'd6) begin
            c7 = t6; c8 = t6;
        end
    end

    // ------------------------------------------------------------------ token flags
    // meta = {hv, hreal, hfirst, hemit, hlast}
    wire rEven  = ~r[0];
    wire vemit  = rEven & (r >= 4) & (r <= H + 2);                     // vertical output row
    wire hflush = r[0] & (r >= 5) & (r <= H + 3) & (x <= 2);           // right-edge flush slot
    wire hv     = vemit | hflush;
    wire hreal  = vemit;
    wire hfirst = vemit & (x == 0);
    wire hemit  = vemit ? ((x >= 4) & ~x[0]) : ~x[0];
    wire hlast  = hflush & (p == NP - 1) & (r == H + 3) & (x == 2);
    wire [4:0] meta = {hv, hreal, hfirst, hemit, hlast};

    // ------------------------------------------------------------------ stage 1: tap registers
    reg signed [7:0] tv0, tv1, tv2, tv3, tv4, tv5, tv6, tv7, tv8;
    reg [4:0] m1, m2, m3, m4, m5, m6;

    always @(posedge aclk) begin
        if (!aresetn) begin
            m1 <= 5'b0;
        end else if (en) begin
            if (go) begin
                tv0 <= c0; tv1 <= c1; tv2 <= c2; tv3 <= c3; tv4 <= c4;
                tv5 <= c5; tv6 <= c6; tv7 <= c7; tv8 <= c8;
                m1  <= meta;
            end else begin
                m1  <= 5'b0;
            end
        end
    end

    // ------------------------------------------------------------------ stages 2-3: vertical FIR
    wire signed [9:0] vy;
    dbwfb2_fir9 #(.IW(8), .OW(10)) u_fv (
        .clk(aclk), .en(en),
        .t0(tv0), .t1(tv1), .t2(tv2), .t3(tv3), .t4(tv4),
        .t5(tv5), .t6(tv6), .t7(tv7), .t8(tv8),
        .y(vy)
    );

    // ------------------------------------------------------------------ stage 4: horizontal window
    reg signed [9:0] w0, w1, w2, w3, w4, w5, w6, w7, w8;

    wire m3_hv    = m3[4];
    wire m3_hreal = m3[3];
    wire m3_hfirst= m3[2];

    always @(posedge aclk) begin
        if (en && m3_hv) begin
            if (m3_hfirst) begin
                w0 <= vy; w1 <= vy; w2 <= vy; w3 <= vy; w4 <= vy;
                w5 <= vy; w6 <= vy; w7 <= vy; w8 <= vy;
            end else begin
                w0 <= m3_hreal ? vy : w0;       // flush: repeat the last real sample
                w1 <= w0; w2 <= w1; w3 <= w2; w4 <= w3;
                w5 <= w4; w6 <= w5; w7 <= w6; w8 <= w7;
            end
        end
    end

    // meta pipeline (reset so no phantom tokens appear after reset)
    always @(posedge aclk) begin
        if (!aresetn) begin
            m2 <= 5'b0; m3 <= 5'b0; m4 <= 5'b0; m5 <= 5'b0; m6 <= 5'b0;
        end else if (en) begin
            m2 <= m1;
            m3 <= m2;
            m4 <= m3;
            m5 <= m4;
            m6 <= m5;
        end
    end

    // ------------------------------------------------------------------ stages 5-6: horizontal FIR
    wire signed [11:0] hy;
    dbwfb2_fir9 #(.IW(10), .OW(12)) u_fh (
        .clk(aclk), .en(en),
        .t0(w0), .t1(w1), .t2(w2), .t3(w3), .t4(w4),
        .t5(w5), .t6(w6), .t7(w7), .t8(w8),
        .y(hy)
    );

    // ------------------------------------------------------------------ stage 7: clip + output register
    wire signed [7:0] hy_clip = (hy > 127) ? 8'sh7F : ((hy < -128) ? 8'sh80 : hy[7:0]);

    reg [7:0] m_data;
    reg       m_last;

    always @(posedge aclk) begin
        if (!aresetn) begin
            m_valid <= 1'b0;
            m_last  <= 1'b0;
        end else if (en) begin
            m_valid <= m6[4] & m6[1];       // hv & hemit
            m_data  <= hy_clip;
            m_last  <= m6[0];               // hlast
        end
    end

    assign m_axis_tdata  = m_data;
    assign m_axis_tvalid = m_valid;
    assign m_axis_tlast  = m_last;
endmodule