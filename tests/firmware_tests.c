#include "drive_guard.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(expr) do { if (!(expr)) { fprintf(stderr,"line %d: %s\n",__LINE__,#expr); exit(1); } } while (0)
static void send_command(rp_guard* g,uint32_t sequence,uint32_t now) {
    rp_command c={sequence,300,200,-100,1};
    uint8_t p[RP_PACKET_SIZE]; rp_encode(&c,p); rp_guard_feed(g,p,sizeof(p),now);
}
int main(void) {
    rp_guard g; rp_command c={1,300,200,-100,1}, decoded;
    uint8_t p[RP_PACKET_SIZE];
    CHECK(rp_crc16((const uint8_t*)"123456789",9)==0x29b1);
    rp_encode(&c,p); CHECK(rp_decode(p,&decoded)); CHECK(decoded.steering_mrad==-100);
    rp_guard_init(&g); rp_guard_feed(&g,p,16,0); CHECK(!g.enabled);
    CHECK(rp_guard_arm(&g,0)); send_command(&g,2,10); CHECK(g.enabled && g.speed_mm_s==300);
    /* Duplicate or old messages must not refresh the deadline. */
    send_command(&g,2,100); send_command(&g,1,150);
    CHECK(g.received_ms==10);
    rp_guard_tick(&g,210,0); CHECK(!g.enabled && !g.armed);
    send_command(&g,3,211); CHECK(!g.enabled);
    CHECK(rp_guard_arm(&g,0)); send_command(&g,4,220); CHECK(g.enabled);
    rp_guard_tick(&g,221,1); CHECK(!g.enabled && g.estop_latched);
    rp_guard_tick(&g,222,0); send_command(&g,5,223); CHECK(!g.enabled);
    CHECK(!rp_guard_arm(&g,1)); CHECK(rp_guard_arm(&g,0));
    /* Late packet arriving before a periodic tick must still trip the watchdog. */
    send_command(&g,6,300); send_command(&g,7,501); CHECK(!g.enabled && !g.armed);
    CHECK(rp_guard_arm(&g,0));
    c.sequence=8; rp_encode(&c,p); p[8]^=1;
    rp_guard_feed(&g,p,16,510); CHECK(!g.enabled);
    c.sequence=9; rp_encode(&c,p);
    rp_guard_feed(&g,p,5,520); CHECK(!g.enabled);
    rp_guard_feed(&g,p+5,11,521); CHECK(g.enabled && g.last_sequence==9);
    /* A partial old frame cannot be completed after a long serial gap. */
    rp_guard_init(&g); CHECK(rp_guard_arm(&g,0));
    c.sequence=1; rp_encode(&c,p); rp_guard_feed(&g,p,8,1);
    rp_guard_feed(&g,p+8,8,100); CHECK(!g.enabled);
    send_command(&g,2,101); CHECK(g.enabled);
    /* Timer and sequence wraparound. */
    rp_guard_init(&g); CHECK(rp_guard_arm(&g,0));
    send_command(&g,0xffffffffu,0xfffffff0u); send_command(&g,0,10); CHECK(g.last_sequence==0);
    rp_guard_tick(&g,210,0); CHECK(!g.enabled);
    /* Reject out-of-range and malformed commands, even with valid CRC. */
    c.sequence=1; c.speed_mm_s=601; rp_encode(&c,p); CHECK(!rp_decode(p,&decoded));
    c.speed_mm_s=300; c.lease_ms=251; rp_encode(&c,p); CHECK(!rp_decode(p,&decoded));
    c.lease_ms=200; c.steering_mrad=601; rp_encode(&c,p); CHECK(!rp_decode(p,&decoded));
    c.steering_mrad=0; c.enable=0; rp_encode(&c,p); CHECK(!rp_decode(p,&decoded));
    c.speed_mm_s=0; rp_encode(&c,p); CHECK(rp_decode(p,&decoded));
    puts("PASS STM32 CRC, parser, bounds, sequencing, watchdog, e-stop, and wraparound");
    return 0;
}
