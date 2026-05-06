"use client";

import {
  Avatar,
  Box,
  Card,
  CardContent,
  Chip,
  Container,
  Stack,
  Typography,
} from "@mui/material";

import NotificationsActiveIcon from "@mui/icons-material/NotificationsActive";
import WorkIcon from "@mui/icons-material/Work";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";
import EventIcon from "@mui/icons-material/Event";

const notifications = [
  {
    id: "1",
    type: "Placement",
    message: "Microsoft hiring drive tomorrow",
    timestamp: "2 mins ago",
    icon: <WorkIcon />,
  },
  {
    id: "2",
    type: "Result",
    message: "Mid exam results published",
    timestamp: "10 mins ago",
    icon: <EmojiEventsIcon />,
  },
  {
    id: "3",
    type: "Event",
    message: "Tech Fest starts tomorrow",
    timestamp: "30 mins ago",
    icon: <EventIcon />,
  },
];

const getChipColor = (type: string) => {
  switch (type) {
    case "Placement":
      return "success";
    case "Result":
      return "primary";
    default:
      return "warning";
  }
};

export default function Home() {
  return (
    <Box
      sx={{
        minHeight: "100vh",
        background:
          "linear-gradient(135deg, #050816 0%, #0f172a 40%, #111827 100%)",
        py: 6,
      }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 5,
            p: 4,
            borderRadius: 5,
            backdropFilter: "blur(16px)",
            background: "rgba(255,255,255,0.08)",
            border: "1px solid rgba(255,255,255,0.15)",
            boxShadow: "0 10px 40px rgba(0,0,0,0.4)",
          }}
        >
          <Box>
            <Typography
              variant="h3"
              fontWeight="bold"
              sx={{
                color: "white",
                mb: 1,
              }}
            >
              Campus Notifications
            </Typography>

            <Typography
              variant="h6"
              sx={{
                color: "rgba(255,255,255,0.7)",
              }}
            >
              Priority Inbox for Students
            </Typography>
          </Box>

          <Avatar
            sx={{
              width: 70,
              height: 70,
              bgcolor: "#2563eb",
              boxShadow: "0 0 30px rgba(37,99,235,0.7)",
            }}
          >
            <NotificationsActiveIcon fontSize="large" />
          </Avatar>
        </Box>

        <Stack spacing={4}>
          {notifications.map((notification) => (
            <Card
              key={notification.id}
              sx={{
                borderRadius: 5,
                overflow: "hidden",
                position: "relative",
                backdropFilter: "blur(20px)",
                background: "rgba(255,255,255,0.08)",
                border: "1px solid rgba(255,255,255,0.15)",
                boxShadow: "0 15px 40px rgba(0,0,0,0.35)",
                transition: "all 0.35s ease",
                transformStyle: "preserve-3d",
                "&:hover": {
                  transform: "translateY(-10px) scale(1.01)",
                  boxShadow: "0 25px 50px rgba(0,0,0,0.5)",
                },
              }}
            >
              <Box
                sx={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  width: "100%",
                  height: "5px",
                  background:
                    notification.type === "Placement"
                      ? "linear-gradient(to right, #22c55e, #4ade80)"
                      : notification.type === "Result"
                      ? "linear-gradient(to right, #3b82f6, #60a5fa)"
                      : "linear-gradient(to right, #f59e0b, #fbbf24)",
                }}
              />

              <CardContent sx={{ p: 4 }}>
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    mb: 3,
                  }}
                >
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 2,
                    }}
                  >
                    <Avatar
                      sx={{
                        bgcolor:
                          notification.type === "Placement"
                            ? "success.main"
                            : notification.type === "Result"
                            ? "primary.main"
                            : "warning.main",
                      }}
                    >
                      {notification.icon}
                    </Avatar>

                    <Chip
                      label={notification.type}
                      color={getChipColor(notification.type)}
                      sx={{
                        fontWeight: "bold",
                        fontSize: "0.9rem",
                        px: 1,
                        py: 2.3,
                      }}
                    />
                  </Box>

                  <Typography
                    variant="body2"
                    sx={{
                      color: "rgba(255,255,255,0.7)",
                      fontWeight: 500,
                    }}
                  >
                    {notification.timestamp}
                  </Typography>
                </Box>

                <Typography
                  variant="h5"
                  fontWeight="bold"
                  sx={{
                    color: "white",
                    mb: 1,
                  }}
                >
                  {notification.message}
                </Typography>

                <Typography
                  variant="body1"
                  sx={{
                    color: "rgba(255,255,255,0.7)",
                  }}
                >
                  Important campus update categorized using priority ranking.
                </Typography>
              </CardContent>
            </Card>
          ))}
        </Stack>
      </Container>
    </Box>
  );
}