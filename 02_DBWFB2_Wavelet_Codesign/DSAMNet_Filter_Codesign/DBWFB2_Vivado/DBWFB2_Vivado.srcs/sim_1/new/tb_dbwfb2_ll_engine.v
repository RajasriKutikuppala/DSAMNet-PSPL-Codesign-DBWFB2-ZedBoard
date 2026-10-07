`timescale 1ns/1ps
// ============================================================================
// Self-checking testbench for dbwfb2_ll_engine.
//   tb_ll_32  : 3 x 32 x 32 , two images back to back, random stalls on both streams (fast)
//   tb_ll_256 : 3 x 256 x 256, one image, random stalls (about 230k clocks)
// Vectors come from reference/dbwfb2_ll_ref.py (folder 'vectors').
// ============================================================================
module tb_core #(
    parameter W = 32,
    parameter H = 32,
    parameter NP = 3,
    parameter NIMG = 2,
    parameter STALLS = 1,
    parameter IN_FILE  = "in.hex",
    parameter EXP_FILE = "exp.hex"
)();
    localparam N_IN  = NP * H * W;
    localparam N_OUT = NP * (H / 2) * (W / 2);

    reg [7:0] in_mem  [0:N_IN-1];
    reg [7:0] exp_mem [0:N_OUT-1];

    reg aclk = 1'b0;
    always #5 aclk = ~aclk;
    reg aresetn = 1'b0;

    reg  [7:0] s_tdata;
    reg        s_tvalid;
    wire       s_tready;
    wire [7:0] m_tdata;
    wire       m_tvalid;
    reg        m_tready;
    wire       m_tlast;

    dbwfb2_ll_engine #(.W(W), .H(H), .NP(NP)) dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
        .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready),
        .m_axis_tlast(m_tlast)
    );

    integer in_idx, in_img, out_idx, out_img, errors, last_errors, cycles;
    reg done;

    initial begin
        $readmemh(IN_FILE,  in_mem);
        $readmemh(EXP_FILE, exp_mem);
        in_idx = 0; in_img = 0; out_idx = 0; out_img = 0;
        errors = 0; last_errors = 0; cycles = 0; done = 1'b0;
        s_tvalid = 1'b0; s_tdata = 8'd0; m_tready = 1'b1;
        repeat (20) @(posedge aclk);
        aresetn = 1'b1;
    end

    // ---------------------------------------------------------------- input driver
    always @(posedge aclk) begin
        if (aresetn) begin
            if (s_tvalid && !s_tready) begin
                // hold the beat until it is accepted
            end else begin
                if (s_tvalid && s_tready) begin
                    if (in_idx == N_IN - 1) begin in_idx = 0; in_img = in_img + 1; end
                    else in_idx = in_idx + 1;
                end
                if (in_img < NIMG && (STALLS == 0 || ($urandom % 10) < 8)) begin
                    s_tvalid <= 1'b1;
                    s_tdata  <= in_mem[in_idx];
                end else begin
                    s_tvalid <= 1'b0;
                end
            end
        end
    end

    // ---------------------------------------------------------------- output checker
    always @(posedge aclk) begin
        if (aresetn && !done) begin
            cycles = cycles + 1;
            if (m_tvalid && m_tready) begin
                if (m_tdata !== exp_mem[out_idx]) begin
                    errors = errors + 1;
                    if (errors <= 10)
                        $display("MISMATCH img %0d idx %0d : got %0d expected %0d",
                                 out_img, out_idx, $signed(m_tdata), $signed(exp_mem[out_idx]));
                end
                if (m_tlast !== (out_idx == N_OUT - 1)) begin
                    last_errors = last_errors + 1;
                    if (last_errors <= 10)
                        $display("TLAST wrong at img %0d idx %0d (tlast=%0b)", out_img, out_idx, m_tlast);
                end
                if (out_idx == N_OUT - 1) begin out_idx = 0; out_img = out_img + 1; end
                else out_idx = out_idx + 1;
            end
            m_tready <= (STALLS == 0) ? 1'b1 : (($urandom % 10) < 7);

            if (out_img == NIMG) begin
                done = 1'b1;
                $display("-----------------------------------------------------------");
                $display("TB %0dx%0dx%0d, %0d image(s): %0d outputs checked, %0d value errors, %0d tlast errors, %0d clocks",
                         NP, H, W, NIMG, NIMG * N_OUT, errors, last_errors, cycles);
                if (errors == 0 && last_errors == 0) $display("RESULT: PASS");
                else                                 $display("RESULT: FAIL");
                $display("-----------------------------------------------------------");
                $finish;
            end else if (cycles > NIMG * N_IN * 40 + 200000) begin
                done = 1'b1;
                $display("RESULT: FAIL (timeout) : only %0d of %0d outputs after %0d clocks",
                         out_img * N_OUT + out_idx, NIMG * N_OUT, cycles);
                $finish;
            end
        end
    end
endmodule

// Top used for the quick run (set this one as the simulation top first)
module tb_ll_32;
    tb_core #(
        .W(32), .H(32), .NP(3), .NIMG(2), .STALLS(1),
        .IN_FILE ("C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet_Filter_Codesign/reference/vectors/in_32.hex"),
        .EXP_FILE("C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet_Filter_Codesign/reference/vectors/exp_32.hex")
    ) t();
endmodule

// Full-size run
module tb_ll_256;
    tb_core #(
        .W(256), .H(256), .NP(3), .NIMG(1), .STALLS(1),
        .IN_FILE ("C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet_Filter_Codesign/reference/vectors/in_256.hex"),
        .EXP_FILE("C:/Users/HP/Desktop/BTP_RemoteSensing/DSAMNet_Filter_Codesign/reference/vectors/exp_256.hex")
    ) t();
endmodule